# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis/power.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

The analytic power functions are normal-approximation formulas, so the central
tests do not re-derive the same algebra. They simulate the actual test statistic
on binomial data under a fixed seed and check that the analytic power matches
the empirical rejection rate, which catches a wrong formula in a way that a
closed-form comparison would not. Published textbook sample sizes are used as
external anchors, and the remaining tests cover monotonicity, boundary
behaviour, round-trip consistency between sizing and reporting, and the input
validation that keeps a mis-specified call from silently returning a number.
"""

from __future__ import annotations

import random
import sys
from math import sqrt
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from power import (  # noqa: E402
    BASELINE_MIR,
    MDE_REGISTERED,
    N_PER_CELL,
    REGISTERED_DESIGN,
    attrition_table,
    equivalence_table,
    holm_planning_alpha,
    mde_two_proportions,
    power_tost_two_proportions,
    power_two_proportions,
    registered_table,
    sample_size_tost,
    sample_size_two_proportions,
)

# --------------------------------------------------------------------------
# Monte Carlo validation of the analytic power function
# --------------------------------------------------------------------------

CRITICAL_Z_TWO_SIDED_05 = 1.959963984540054
CRITICAL_Z_TWO_SIDED_025 = 2.241402727604947


def _pooled_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """The two-proportion z statistic the power function is derived for."""
    p1, p2 = k1 / n1, k2 / n2
    p_bar = (k1 + k2) / (n1 + n2)
    se = sqrt(p_bar * (1.0 - p_bar) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0
    return (p2 - p1) / se


def _empirical_power(
    p1: float, p2: float, n: int, z_crit: float, reps: int, seed: int
) -> float:
    rng = random.Random(seed)
    rejections = 0
    for _ in range(reps):
        k1 = sum(rng.random() < p1 for _ in range(n))
        k2 = sum(rng.random() < p2 for _ in range(n))
        if abs(_pooled_z(k1, n, k2, n)) >= z_crit:
            rejections += 1
    return rejections / reps


@pytest.mark.parametrize(
    "p1, p2, n, alpha, z_crit, seed",
    [
        (0.10, 0.20, 100, 0.05, CRITICAL_Z_TWO_SIDED_05, 20261025),
        (0.10, 0.20, 300, 0.05, CRITICAL_Z_TWO_SIDED_05, 20261026),
        (0.10, 0.20, 100, 0.025, CRITICAL_Z_TWO_SIDED_025, 20261027),
        (0.20, 0.35, 150, 0.05, CRITICAL_Z_TWO_SIDED_05, 20261028),
        (0.05, 0.15, 200, 0.05, CRITICAL_Z_TWO_SIDED_05, 20261029),
    ],
)
def test_analytic_power_matches_simulation(p1, p2, n, alpha, z_crit, seed):
    analytic = power_two_proportions(p1, p2, n, alpha=alpha)
    empirical = _empirical_power(p1, p2, n, z_crit, reps=4000, seed=seed)
    # Monte Carlo standard error at 4000 reps is at most 0.008; 0.03 leaves room
    # for that plus the discreteness the normal approximation smooths over.
    assert abs(analytic - empirical) < 0.03, (
        f"analytic {analytic:.3f} vs empirical {empirical:.3f} "
        f"for p1={p1}, p2={p2}, n={n}, alpha={alpha}"
    )


def test_type_i_error_under_the_null_matches_alpha():
    """With no true difference, rejection rate must sit at alpha, not above it."""
    empirical = _empirical_power(
        0.10, 0.10, 400, CRITICAL_Z_TWO_SIDED_05, reps=4000, seed=20261030
    )
    assert abs(empirical - 0.05) < 0.02
    assert power_two_proportions(0.10, 0.10, 400, alpha=0.05) == pytest.approx(
        0.05, abs=1e-9
    )


# --------------------------------------------------------------------------
# External anchors from the published literature
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p1, p2, expected_n",
    [
        (0.05, 0.10, 435),  # Fleiss, uncorrected, 80% power, alpha 0.05
        (0.60, 0.80, 81),
        (0.20, 0.40, 82),
    ],
)
def test_sample_size_matches_published_anchors(p1, p2, expected_n):
    n = sample_size_two_proportions(p1, p2, alpha=0.05, power=0.80)
    # Published tables differ by a unit or two depending on rounding and whether
    # a continuity correction is applied; the formula itself is what is checked.
    assert abs(n - expected_n) <= 2, f"got {n}, expected about {expected_n}"


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------


def test_power_increases_with_sample_size():
    powers = [power_two_proportions(0.10, 0.20, n, alpha=0.05) for n in (50, 100, 200, 400, 800)]
    assert powers == sorted(powers)
    assert powers[0] < powers[-1]


def test_power_increases_with_effect_size():
    powers = [power_two_proportions(0.10, p2, 200, alpha=0.05) for p2 in (0.12, 0.15, 0.20, 0.30)]
    assert powers == sorted(powers)


def test_power_decreases_as_alpha_tightens():
    loose = power_two_proportions(0.10, 0.20, 200, alpha=0.05)
    tight = power_two_proportions(0.10, 0.20, 200, alpha=0.0125)
    assert tight < loose


def test_one_sided_power_exceeds_two_sided_in_the_hypothesised_direction():
    two = power_two_proportions(0.10, 0.20, 200, alpha=0.05, alternative="two-sided")
    one = power_two_proportions(0.10, 0.20, 200, alpha=0.05, alternative="greater")
    assert one > two


def test_one_sided_power_collapses_in_the_wrong_direction():
    """A directional comparison must not be rewarded by an effect that reverses."""
    wrong_way = power_two_proportions(0.20, 0.10, 200, alpha=0.05, alternative="greater")
    assert wrong_way < 0.01


def test_power_is_symmetric_under_arm_swap_for_two_sided_tests():
    a = power_two_proportions(0.10, 0.25, 180, alpha=0.05)
    b = power_two_proportions(0.25, 0.10, 180, alpha=0.05)
    assert a == pytest.approx(b, abs=1e-12)


def test_unequal_arms_are_weaker_than_equal_arms_at_the_same_total_n():
    equal = power_two_proportions(0.10, 0.20, 300, 300, alpha=0.05)
    unequal = power_two_proportions(0.10, 0.20, 100, 500, alpha=0.05)
    assert unequal < equal


# --------------------------------------------------------------------------
# Round-trip consistency: sizing, reporting and MDE must agree
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.025, 0.0125])
@pytest.mark.parametrize("target", [0.80, 0.90])
def test_sample_size_is_the_smallest_n_reaching_target(alpha, target):
    n = sample_size_two_proportions(BASELINE_MIR, BASELINE_MIR + MDE_REGISTERED, alpha, target)
    assert power_two_proportions(BASELINE_MIR, BASELINE_MIR + MDE_REGISTERED, n, alpha=alpha) >= target
    assert power_two_proportions(BASELINE_MIR, BASELINE_MIR + MDE_REGISTERED, n - 1, alpha=alpha) < target


def test_mde_round_trips_through_the_power_function():
    mde = mde_two_proportions(BASELINE_MIR, N_PER_CELL, alpha=0.0125, power=0.80)
    at_mde = power_two_proportions(BASELINE_MIR, BASELINE_MIR + mde, N_PER_CELL, alpha=0.0125)
    below = power_two_proportions(BASELINE_MIR, BASELINE_MIR + mde - 0.01, N_PER_CELL, alpha=0.0125)
    assert at_mde == pytest.approx(0.80, abs=1e-3)
    assert below < 0.80


def test_mde_shrinks_as_sample_size_grows():
    mdes = [mde_two_proportions(BASELINE_MIR, n, alpha=0.05) for n in (100, 300, 1000)]
    assert mdes == sorted(mdes, reverse=True)


# --------------------------------------------------------------------------
# Holm planning alpha
# --------------------------------------------------------------------------


def test_holm_planning_alpha_uses_the_most_stringent_step():
    assert holm_planning_alpha(0.05, 1) == pytest.approx(0.05)
    assert holm_planning_alpha(0.05, 2) == pytest.approx(0.025)
    assert holm_planning_alpha(0.05, 4) == pytest.approx(0.0125)


def test_holm_planning_alpha_rejects_zero_comparisons():
    with pytest.raises(ValueError):
        holm_planning_alpha(0.05, 0)


# --------------------------------------------------------------------------
# The registered design: the findings this module exists to record
# --------------------------------------------------------------------------


def _cell(prefix: str):
    return next(c for c in REGISTERED_DESIGN if c.label.startswith(prefix))


def test_h1_is_adequately_powered_when_pooled():
    """H1 is the core claim and the registered n carries it comfortably."""
    assert _cell("H1 pooled").power > 0.99


def test_h2_as_registered_is_underpowered():
    """100 runs per cell cannot see the registered minimum effect of interest.

    This is the finding that motivates the v0.2 revision: at alpha 0.0125 the
    per-category comparison has roughly a 30% chance of detecting a true 0.10
    absolute difference, so F2 would fire on most runs of a real effect.
    """
    h2 = _cell("H2 per category, as registered")
    assert h2.n_per_arm == N_PER_CELL
    assert h2.power < 0.35
    assert h2.n_for_80 > 2.5 * N_PER_CELL


def test_h2_pooled_over_model_families_recovers_power():
    assert _cell("H2 per category, pooled over model families").power > 0.80


def test_per_cell_mde_is_far_above_the_registered_effect_of_interest():
    """At n=100 the smallest visible effect is nearly twice what was registered."""
    h2 = _cell("H2 per category, as registered")
    assert h2.mde > 1.5 * MDE_REGISTERED


def test_power_survives_the_worst_discard_rate_f4_permits_for_h1():
    """F4 tolerates 30% discard under D6, so H1 must hold at 70 valid runs/cell."""
    valid = int(N_PER_CELL * 0.70)
    h1 = _cell("H1 pooled")
    pooled = valid * 5 * 3
    assert power_two_proportions(BASELINE_MIR, BASELINE_MIR + MDE_REGISTERED, pooled, alpha=h1.alpha) > 0.95


def test_h2_does_not_survive_the_worst_discard_rate():
    valid = int(N_PER_CELL * 0.70)
    h2 = _cell("H2 per category, as registered")
    assert power_two_proportions(BASELINE_MIR, BASELINE_MIR + MDE_REGISTERED, valid, alpha=h2.alpha) < 0.25


# --------------------------------------------------------------------------
# Equivalence testing, for restating F2 as a positive claim
# --------------------------------------------------------------------------


def test_tost_type_i_error_is_bounded_outside_the_margin():
    """Beyond the margin the procedure must not declare equivalence.

    At exactly the margin TOST's rejection rate is bounded by alpha; past it the
    rate falls toward zero. Asserting a hard zero would be asserting more than
    the procedure guarantees.
    """
    alpha = 0.0125
    at_margin = power_tost_two_proportions(0.10, 0.15, 1000, margin=0.05, alpha=alpha)
    beyond = power_tost_two_proportions(0.10, 0.20, 1000, margin=0.05, alpha=alpha)
    assert 0.0 <= at_margin <= alpha
    assert beyond < at_margin
    assert beyond < 1e-4


def test_tost_power_grows_with_sample_size_inside_the_margin():
    powers = [power_tost_two_proportions(0.10, 0.10, n, margin=0.05, alpha=0.0125) for n in (200, 600, 1200, 2400)]
    assert powers == sorted(powers)
    assert powers[-1] > 0.90


def test_tost_power_falls_as_the_true_difference_approaches_the_margin():
    powers = [power_tost_two_proportions(0.10, 0.10 + d, 1500, margin=0.05, alpha=0.0125) for d in (0.0, 0.01, 0.02, 0.03)]
    assert powers == sorted(powers, reverse=True)


def test_equivalence_at_the_registered_n_is_not_achievable():
    """F2 cannot be restated as equivalence without a far larger design.

    At 100 or 300 runs per arm the equivalence procedure has essentially no
    power at a 0.05 margin, which is the quantitative reason F2 has to either
    grow the sample or widen the margin rather than keep its current wording.
    """
    assert power_tost_two_proportions(BASELINE_MIR, BASELINE_MIR, 100, margin=0.05, alpha=0.0125) < 0.01
    assert power_tost_two_proportions(BASELINE_MIR, BASELINE_MIR, 300, margin=0.05, alpha=0.0125) < 0.05
    assert sample_size_tost(BASELINE_MIR, BASELINE_MIR, margin=0.05, alpha=0.0125, power=0.80) > 800


def test_sample_size_tost_round_trips():
    n = sample_size_tost(0.10, 0.10, margin=0.10, alpha=0.0125, power=0.80)
    assert power_tost_two_proportions(0.10, 0.10, n, margin=0.10, alpha=0.0125) >= 0.80
    assert power_tost_two_proportions(0.10, 0.10, n - 1, margin=0.10, alpha=0.0125) < 0.80


def test_sample_size_tost_refuses_a_difference_outside_the_margin():
    with pytest.raises(ValueError):
        sample_size_tost(0.10, 0.20, margin=0.05)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p1": -0.01, "p2": 0.2, "n1": 100},
        {"p1": 0.1, "p2": 1.5, "n1": 100},
        {"p1": 0.1, "p2": 0.2, "n1": 0},
        {"p1": 0.1, "p2": 0.2, "n1": 100, "n2": -5},
        {"p1": 0.1, "p2": 0.2, "n1": 100, "alpha": 0.0},
        {"p1": 0.1, "p2": 0.2, "n1": 100, "alpha": 1.0},
        {"p1": 0.1, "p2": 0.2, "n1": 100, "alternative": "sideways"},
    ],
)
def test_power_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        power_two_proportions(**kwargs)


def test_sample_size_rejects_a_zero_effect():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.10, 0.10, alpha=0.05, power=0.80)


def test_sample_size_rejects_an_impossible_power():
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.10, 0.20, alpha=0.05, power=1.0)


def test_mde_rejects_an_n_too_small_to_detect_anything():
    with pytest.raises(ValueError):
        mde_two_proportions(0.10, 2, alpha=0.0125, power=0.80)


def test_boundary_proportions_do_not_divide_by_zero():
    assert power_two_proportions(0.0, 0.0, 100, alpha=0.05) == 0.0
    assert power_two_proportions(0.0, 1.0, 100, alpha=0.05) == 1.0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_tables_render_every_registered_cell():
    table = registered_table()
    for cell in REGISTERED_DESIGN:
        assert cell.label in table
    assert "0.10" in table


def test_attrition_table_covers_the_f4_ceiling():
    table = attrition_table()
    assert "30%" in table
    assert "70" in table


def test_equivalence_table_renders():
    assert "TOST" in equivalence_table(margin=0.05)
