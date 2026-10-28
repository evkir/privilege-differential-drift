# SPDX-License-Identifier: Apache-2.0

"""Power, sample size and minimum detectable effect for the registered design.

Implements the frozen power analysis referenced in PREREGISTRATION.md §5
("Power analysis assuming a baseline MIR of 0.10 and a minimum effect of
interest of 0.10 absolute difference is to be completed and recorded before
collection begins").

Scope: all primary and secondary comparisons in this pre-registration reduce to
a contrast between two independent binomial proportions (MIR is a run-level
proportion; a run either contains a mandate inflation event or it does not).
The mixed-effects logistic model in §5 is the analysis model; this module sizes
the design against the marginal two-proportion contrasts that the pre-specified
comparisons in §4 are stated in. Treating the design this way is conservative
for H1 (the regression pools information across cells) and exact for H2, where
each comparison is a single category against the control category.

Multiplicity: Holm is stated in §5. For *planning* purposes the relevant alpha
is the most stringent step of the Holm sequence, alpha / k, because a design
powered only at the least stringent step is not powered for the comparison that
has to clear the first hurdle. `holm_planning_alpha` returns that value.

Equivalence: F2 is written as a failure to reject, which at low power is
absence of evidence rather than evidence of absence. `power_tost_two_proportions`
sizes the two-one-sided-tests procedure that turns F2 into a positive claim of
equivalence within a stated margin.

Methods are normal-approximation (score-type statistic with pooled variance
under the null, unpooled under the alternative), which is the standard planning
approximation for proportions in this range. `tests/test_power.py` validates the
analytic results against Monte Carlo simulation of the same test statistic.

No external dependencies: standard library only, so the pre-registration's
analysis gate runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

__all__ = [
    "holm_planning_alpha",
    "power_two_proportions",
    "sample_size_two_proportions",
    "mde_two_proportions",
    "power_tost_two_proportions",
    "sample_size_tost",
    "Cell",
    "REGISTERED_DESIGN",
    "registered_table",
]

_N = NormalDist()

#: Baseline MIR assumed in PREREGISTRATION.md §5.
BASELINE_MIR = 0.10
#: Minimum effect of interest, absolute difference in MIR, §5.
MDE_REGISTERED = 0.10
#: Registered collection target per cell, §5.
N_PER_CELL = 135
#: Discard rate F4 tolerates under atomic benignness (D6), §6.
DISCARD_CEILING = 0.30
#: Substantive + control data categories, §5.
N_CATEGORIES = 5
#: Minimum model families from distinct providers, §5.
N_MODEL_FAMILIES = 3


def _critical_z(alpha: float, alternative: str) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative == "two-sided":
        return _N.inv_cdf(1.0 - alpha / 2.0)
    if alternative in ("greater", "less", "one-sided"):
        return _N.inv_cdf(1.0 - alpha)
    raise ValueError(f"unknown alternative: {alternative!r}")


def _check_proportion(name: str, p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a proportion in [0, 1], got {p}")


def _check_n(name: str, n: int) -> None:
    if n < 1:
        raise ValueError(f"{name} must be a positive integer, got {n}")


def holm_planning_alpha(alpha: float, n_comparisons: int) -> float:
    """Most stringent alpha in a Holm sequence of ``n_comparisons`` tests.

    Holm tests the smallest p-value against alpha / k. A design powered only at
    the final, least stringent step is not powered for the comparison that must
    clear the first step, so planning uses alpha / k.
    """
    if n_comparisons < 1:
        raise ValueError(f"n_comparisons must be >= 1, got {n_comparisons}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha / n_comparisons


def power_two_proportions(
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Power to detect a difference between two independent proportions.

    ``p1`` is the reference (control / smaller gap) proportion, ``p2`` the
    alternative. ``n2`` defaults to ``n1``. Returns power in [0, 1].

    For ``alternative="greater"`` the alternative hypothesis is p2 > p1; the
    returned power is near zero if p2 < p1, which is the intended behaviour for
    a directional pre-specified comparison.
    """
    _check_proportion("p1", p1)
    _check_proportion("p2", p2)
    _check_n("n1", n1)
    n2 = n1 if n2 is None else n2
    _check_n("n2", n2)

    delta = p2 - p1
    p_bar = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_null = sqrt(p_bar * (1.0 - p_bar) * (1.0 / n1 + 1.0 / n2))
    se_alt = sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)

    if se_alt == 0.0:
        # Degenerate: both proportions at a boundary. Any nonzero difference is
        # detected with certainty, no difference is never detected.
        return 1.0 if delta != 0.0 else 0.0

    z = _critical_z(alpha, alternative)

    if alternative == "two-sided":
        upper = _N.cdf((delta - z * se_null) / se_alt)
        lower = _N.cdf((-delta - z * se_null) / se_alt)
        return upper + lower
    if alternative in ("greater", "one-sided"):
        return _N.cdf((delta - z * se_null) / se_alt)
    return _N.cdf((-delta - z * se_null) / se_alt)


def sample_size_two_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    max_n: int = 10_000_000,
) -> int:
    """Smallest equal per-group n reaching ``power``.

    Exact integer search (doubling then bisection) over the same power function
    used for reporting, so sizing and reporting cannot drift apart.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ to size a study")

    def reached(n: int) -> bool:
        return power_two_proportions(p1, p2, n, n, alpha, alternative) >= power

    lo, hi = 1, 1
    while not reached(hi):
        lo = hi
        hi *= 2
        if hi > max_n:
            raise ValueError(
                f"required n exceeds max_n={max_n} for p1={p1}, p2={p2}, "
                f"alpha={alpha}, power={power}"
            )
    while lo < hi:
        mid = (lo + hi) // 2
        if reached(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def mde_two_proportions(
    p1: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    tol: float = 1e-9,
) -> float:
    """Smallest detectable p2 - p1 (p2 > p1) at the given n, alpha and power.

    Returns the absolute difference. Raises if no p2 <= 1 reaches ``power``.
    """
    _check_proportion("p1", p1)
    _check_n("n", n)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")

    if power_two_proportions(p1, 1.0, n, n, alpha, alternative) < power:
        raise ValueError(
            f"no detectable effect at n={n}, alpha={alpha}, power={power} "
            f"with baseline p1={p1}"
        )

    lo, hi = p1, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if power_two_proportions(p1, mid, n, n, alpha, alternative) >= power:
            hi = mid
        else:
            lo = mid
    return hi - p1


def power_tost_two_proportions(
    p1: float,
    p2: float,
    n1: int,
    n2: int | None = None,
    margin: float = 0.05,
    alpha: float = 0.05,
) -> float:
    """Power of two one-sided tests for equivalence within +/- ``margin``.

    Used to size F2 as a positive equivalence claim rather than a failure to
    reject. ``margin`` is on the absolute difference in proportions.

    When |p2 - p1| >= margin the true state is non-equivalence, and the returned
    value is the procedure's type I error rather than power: TOST holds it at or
    below ``alpha`` at the margin and drives it toward zero beyond it. The
    result is clipped at zero so the approximation cannot return a negative.
    """
    _check_proportion("p1", p1)
    _check_proportion("p2", p2)
    _check_n("n1", n1)
    n2 = n1 if n2 is None else n2
    _check_n("n2", n2)
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")

    se = sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    if se == 0.0:
        return 1.0 if abs(p2 - p1) < margin else 0.0

    z = _N.inv_cdf(1.0 - alpha)
    delta = abs(p2 - p1)
    lower = _N.cdf((margin - delta) / se - z)
    upper = _N.cdf((margin + delta) / se - z)
    return max(0.0, lower + upper - 1.0)


def sample_size_tost(
    p1: float,
    p2: float,
    margin: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
    max_n: int = 10_000_000,
) -> int:
    """Smallest equal per-group n for equivalence power within ``margin``."""
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if abs(p2 - p1) >= margin:
        raise ValueError(
            f"true difference {abs(p2 - p1)} is outside the equivalence margin "
            f"{margin}; no sample size establishes equivalence"
        )

    def reached(n: int) -> bool:
        return power_tost_two_proportions(p1, p2, n, n, margin, alpha) >= power

    lo, hi = 1, 1
    while not reached(hi):
        lo = hi
        hi *= 2
        if hi > max_n:
            raise ValueError(f"required n exceeds max_n={max_n}")
    while lo < hi:
        mid = (lo + hi) // 2
        if reached(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True)
class Cell:
    """One pre-specified comparison, with the n the registered design gives it."""

    label: str
    hypothesis: str
    n_per_arm: int
    n_comparisons: int
    alpha_family: float = 0.05
    baseline: float = BASELINE_MIR
    effect: float = MDE_REGISTERED

    @property
    def alpha(self) -> float:
        return holm_planning_alpha(self.alpha_family, self.n_comparisons)

    @property
    def power(self) -> float:
        return power_two_proportions(
            self.baseline, self.baseline + self.effect, self.n_per_arm,
            alpha=self.alpha,
        )

    @property
    def n_for_80(self) -> int:
        return sample_size_two_proportions(
            self.baseline, self.baseline + self.effect, self.alpha, 0.80
        )

    @property
    def mde(self) -> float:
        return mde_two_proportions(self.baseline, self.n_per_arm, self.alpha, 0.80)


#: H1 states two pre-specified comparisons (§4): MIR(0→2) > MIR(0→1) and
#: MIR(0→2) > MIR(1→2). H2 states one comparison per non-control category
#: against the control (§4, §5: four substantive categories).
H1_COMPARISONS = 2
H2_COMPARISONS = N_CATEGORIES - 1

REGISTERED_DESIGN: tuple[Cell, ...] = (
    Cell(
        "H1 pooled over categories and model families",
        "H1", N_PER_CELL * N_CATEGORIES * N_MODEL_FAMILIES, H1_COMPARISONS,
    ),
    Cell(
        "H1 within one model family",
        "H1", N_PER_CELL * N_CATEGORIES, H1_COMPARISONS,
    ),
    Cell(
        "H1 within a single cell",
        "H1", N_PER_CELL, H1_COMPARISONS,
    ),
    Cell(
        "H2 per category, single cell (descriptive only)",
        "H2", N_PER_CELL, H2_COMPARISONS,
    ),
    Cell(
        "H2 per category, pooled over model families",
        "H2", N_PER_CELL * N_MODEL_FAMILIES, H2_COMPARISONS,
    ),
    Cell(
        "H2 per category, pooled, at the F4 discard ceiling",
        "H2", int(N_PER_CELL * N_MODEL_FAMILIES * (1.0 - DISCARD_CEILING)), H2_COMPARISONS,
    ),
)


def registered_table() -> str:
    """Rendered power table for the registered design."""
    header = (
        f"{'comparison':46s} {'n/arm':>7s} {'alpha':>7s} {'power':>7s} "
        f"{'n for 0.80':>11s} {'MDE':>7s}"
    )
    lines = [
        f"baseline MIR {BASELINE_MIR:.2f}, effect of interest "
        f"{MDE_REGISTERED:+.2f} absolute, two-sided, Holm planning alpha",
        "",
        header,
        "-" * len(header),
    ]
    for cell in REGISTERED_DESIGN:
        lines.append(
            f"{cell.label:46s} {cell.n_per_arm:7d} {cell.alpha:7.4f} "
            f"{cell.power:7.3f} {cell.n_for_80:11d} {cell.mde:7.3f}"
        )
    return "\n".join(lines)


def attrition_table(discard_rates: tuple[float, ...] = (0.0, 0.15, 0.30)) -> str:
    """Power after runs are discarded under atomic benignness (D6).

    F4 permits a discard rate up to 30% before the premise is considered failed,
    so the design has to hold at the discounted figure, not the collection target.
    """
    h1 = next(c for c in REGISTERED_DESIGN if c.label.startswith("H1 pooled"))
    h2 = next(c for c in REGISTERED_DESIGN if c.label.startswith("H2 per category, single cell"))
    header = f"{'discard':>8s} {'valid/cell':>11s} {'H1 pooled':>10s} {'H2 per cell':>12s}"
    lines = [header, "-" * len(header)]
    for rate in discard_rates:
        valid = int(round(N_PER_CELL * (1.0 - rate)))
        lines.append(
            f"{rate:7.0%} {valid:11d} "
            f"{power_two_proportions(h1.baseline, h1.baseline + h1.effect, valid * N_CATEGORIES * N_MODEL_FAMILIES, alpha=h1.alpha):10.3f} "
            f"{power_two_proportions(h2.baseline, h2.baseline + h2.effect, valid, alpha=h2.alpha):12.3f}"
        )
    return "\n".join(lines)


def equivalence_table(margin: float = 0.05) -> str:
    """Sizing for F2 restated as an equivalence test within ``margin``."""
    alpha = holm_planning_alpha(0.05, H2_COMPARISONS)
    lines = [
        f"F2 as equivalence (TOST), margin {margin:+.2f} absolute, "
        f"alpha {alpha:.4f} per one-sided test",
        "",
        f"{'true difference':>16s} {'n/arm = 100':>12s} {'n/arm = 300':>12s} {'n for 0.80':>11s}",
        "-" * 55,
    ]
    for true_diff in (0.0, 0.01, 0.02):
        p2 = BASELINE_MIR + true_diff
        try:
            n80 = str(sample_size_tost(BASELINE_MIR, p2, margin, alpha, 0.80))
        except ValueError:
            n80 = "n/a"
        lines.append(
            f"{true_diff:16.2f} "
            f"{power_tost_two_proportions(BASELINE_MIR, p2, 100, margin=margin, alpha=alpha):12.3f} "
            f"{power_tost_two_proportions(BASELINE_MIR, p2, 300, margin=margin, alpha=alpha):12.3f} "
            f"{n80:>11s}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(registered_table())
    print()
    print(attrition_table())
    print()
    print(equivalence_table(margin=0.05))
    print()
    print(equivalence_table(margin=0.10))
