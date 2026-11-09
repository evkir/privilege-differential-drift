# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis/model.py.

Written by Claude (Opus 5, Anthropic) at the author's request.

`analysis/model.py` implements the registered estimator rather than calling a
library for it, so the burden of proof is higher than for a module that wraps
something already trusted. Three independent checks carry it, and none of them
re-derives the module's own algebra:

* **Against a different quadrature.** The marginal likelihood is evaluated a
  second time on a dense fixed grid, which shares no code with the adaptive
  scheme — different nodes, different weights, no mode-finding — and the two
  must agree to within a stated tolerance.
* **Against a different model.** With every run in its own group the random
  effect is unidentified, so the fit must reproduce the ordinary logistic
  maximum likelihood that scipy finds independently.
* **Against the sampling distribution.** Parameter recovery, Wald interval
  coverage and type I error are measured by simulating from the model the
  estimator assumes, in the manner `analysis/power.py` is already validated.
  A formula comparison would pass on a wrong implementation of the right
  formula; a rejection rate will not.

The remaining tests cover the record schema, which is the boundary where a
collection fault has to be caught, the design matrix coding the registered
contrasts depend on, and the §6 criteria, each exercised on data constructed to
make exactly one of them fire.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from allowlists import GAP_CONDITIONS  # noqa: E402
from model import (  # noqa: E402
    AIR_F3_THRESHOLD,
    ALPHA_FAMILY,
    EQUIVALENCE_MARGIN,
    QUADRATURE_NODES,
    Design,
    RunRecord,
    _group_log_marginal,
    attribution_identification_rate,
    bootstrap_cell_ci,
    build_design,
    discard_rate,
    evaluate_falsification,
    fit_random_intercept_logit,
    h1_contrasts,
    h2_contrasts,
    holm,
    load_runs,
    parse_runs,
    per_category_mir,
    per_cell_mir,
    report,
    tost_equivalence,
    valid_runs,
    wald_contrast,
)
from power import H2_COMPARISONS, MDE_REGISTERED, holm_planning_alpha  # noqa: E402
from tasks import TASKS, Category  # noqa: E402

# Monte Carlo sizes. Each fit costs tens of milliseconds, and this suite gates
# every commit, so the replication counts are the smallest that still make the
# assertions meaningful: at 250 replications the standard error on a rejection
# rate near 0.05 is about 0.014, and the tolerances below are set at roughly
# three of those.
_NULL_REPLICATIONS = 250
_ALT_REPLICATIONS = 60
_MC_GROUPS = 10
_MC_PER_GROUP = 24
_MC_NODES = 9

_TASKS_BY_CATEGORY = {
    category: tuple(t for t in TASKS if t.category is category) for category in Category
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate(
    rng: np.random.Generator,
    *,
    n_groups: int,
    per_group: int,
    beta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One draw from the model the estimator assumes."""
    groups = np.repeat(np.arange(n_groups), per_group)
    n = len(groups)
    x = (rng.random(n) < 0.5).astype(float)
    X = np.column_stack([np.ones(n), x])
    u = rng.normal(0.0, sigma, size=n_groups)[groups]
    p = 1.0 / (1.0 + np.exp(-(X @ beta + u)))
    y = (rng.random(n) < p).astype(float)
    return y, X, groups


def _dense_log_marginal(
    eta: np.ndarray, y: np.ndarray, sigma: float, lo: float = -12.0, hi: float = 12.0
) -> float:
    """The same integral on a dense fixed grid, sharing no code with the module.

    No mode-finding, no Gauss-Hermite nodes, no curvature scaling: a trapezoid
    over a wide interval. Slow, and that is the point — it is only ever used to
    check the fast one.
    """
    u = np.linspace(lo, hi, 20001)
    e = eta[:, None] + sigma * u[None, :]
    loglik = (y[:, None] * e - np.logaddexp(0.0, e)).sum(axis=0)
    log_phi = -0.5 * u**2 - 0.5 * math.log(2.0 * math.pi)
    terms = loglik + log_phi
    peak = float(terms.max())
    return peak + float(np.log(np.trapezoid(np.exp(terms - peak), u)))


def _plain_logistic_mle(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary logistic MLE, found by scipy without reference to the module."""

    def negative_loglik(beta: np.ndarray) -> float:
        eta = X @ beta
        return float((np.logaddexp(0.0, eta) - y * eta).sum())

    return optimize.minimize(
        negative_loglik, np.zeros(X.shape[1]), method="BFGS"
    ).x


def _record(**overrides: object) -> RunRecord:
    """A valid record, with fields overridden by keyword."""
    task = _TASKS_BY_CATEGORY[Category.FINANCIAL][0]
    fields: dict[str, object] = {
        "run_id": "r1",
        "task_id": task.task_id,
        "category": task.category,
        "writer_level": 0,
        "consumer_level": 1,
        "model_family": "alpha",
        "valid": True,
        "inflated": False,
    }
    fields.update(overrides)
    return RunRecord(**fields)  # type: ignore[arg-type]


def _synthetic_runs(
    *,
    seed: int,
    effect_02: float = 0.0,
    effect_12: float = 0.0,
    category_effect: float = 0.0,
    discard: float = 0.0,
    attribution: float = 1.0,
    n_per_cell: int = 60,
    families: tuple[str, ...] = ("alpha", "beta", "gamma"),
    intercept: float = -2.0,
    sigma_task: float = 0.4,
) -> tuple[RunRecord, ...]:
    """A full crossing of the registered design, with effects set by argument."""
    rng = np.random.default_rng(seed)
    task_effect = {task.task_id: rng.normal(0.0, sigma_task) for task in TASKS}
    records: list[RunRecord] = []
    index = 0
    for gap in GAP_CONDITIONS:
        for category in Category:
            tasks = _TASKS_BY_CATEGORY[category]
            for family in families:
                for k in range(n_per_cell // len(families)):
                    task = tasks[k % len(tasks)]
                    eta = intercept + task_effect[task.task_id]
                    if not category.is_control:
                        eta += category_effect
                    if gap == (0, 2):
                        eta += effect_02
                    elif gap == (1, 2):
                        eta += effect_12
                    inflated = bool(rng.random() < 1.0 / (1.0 + math.exp(-eta)))
                    kept = rng.random() >= discard
                    index += 1
                    events = 1 if (inflated and kept) else 0
                    records.append(
                        RunRecord(
                            run_id=f"r{index:05d}",
                            task_id=task.task_id,
                            category=category,
                            writer_level=gap[0],
                            consumer_level=gap[1],
                            model_family=family,
                            valid=kept,
                            inflated=inflated and kept,
                            discard_reason=None if kept else "d6_review",
                            n_inflation_events=events,
                            n_events_attributed=(
                                events if (events and rng.random() < attribution) else 0
                            ),
                        )
                    )
    return tuple(records)


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


def test_record_accepts_a_registered_combination() -> None:
    record = _record()
    assert record.gap in GAP_CONDITIONS
    assert record.gap_label == "0->1"
    assert 0.0 < record.normalized_gap < 1.0


def test_record_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task_id"):
        _record(task_id="not-a-task")


def test_record_rejects_category_disagreeing_with_the_frozen_task_set() -> None:
    """The category is a property of the task, so a record cannot restate it."""
    with pytest.raises(ValueError, match="is registered in category"):
        _record(category=Category.CONTROL)


@pytest.mark.parametrize("gap", [(0, 0), (1, 0), (2, 2), (2, 0), (1, 1)])
def test_record_rejects_unregistered_gap_conditions(gap: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="registered gap conditions"):
        _record(writer_level=gap[0], consumer_level=gap[1])


def test_record_rejects_empty_model_family() -> None:
    with pytest.raises(ValueError, match="model_family is empty"):
        _record(model_family="")


def test_record_rejects_a_valid_run_carrying_a_discard_reason() -> None:
    with pytest.raises(ValueError, match="carries no discard reason"):
        _record(valid=True, discard_reason="d6_review")


def test_record_rejects_a_discarded_run_without_a_reason() -> None:
    with pytest.raises(ValueError, match="must say why"):
        _record(valid=False)


def test_record_rejects_inflation_without_an_event() -> None:
    with pytest.raises(ValueError, match="no inflation event is counted"):
        _record(inflated=True, n_inflation_events=0)


def test_record_rejects_events_on_a_run_not_marked_inflated() -> None:
    with pytest.raises(ValueError, match="not marked inflated"):
        _record(inflated=False, n_inflation_events=2)


def test_record_rejects_more_attributions_than_events() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _record(inflated=True, n_inflation_events=1, n_events_attributed=2)


def test_record_rejects_more_depths_than_events() -> None:
    with pytest.raises(ValueError, match="more depths recorded"):
        _record(inflated=True, n_inflation_events=1, inflation_depths=(1, 2))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _line(record: RunRecord) -> str:
    return json.dumps(
        {
            "run_id": record.run_id,
            "task_id": record.task_id,
            "category": record.category.value,
            "writer_level": record.writer_level,
            "consumer_level": record.consumer_level,
            "model_family": record.model_family,
            "valid": record.valid,
            "inflated": record.inflated,
            "discard_reason": record.discard_reason,
            "n_inflation_events": record.n_inflation_events,
            "n_events_attributed": record.n_events_attributed,
        }
    )


def test_parse_runs_round_trips() -> None:
    records = _synthetic_runs(seed=1, n_per_cell=3)
    parsed = parse_runs(_line(r) for r in records)
    assert len(parsed) == len(records)
    assert parsed[0].task_id == records[0].task_id
    assert parsed[0].category is records[0].category


def test_parse_runs_skips_blank_lines_and_comments() -> None:
    record = _record()
    parsed = parse_runs(["", "   ", "# a note", _line(record), ""])
    assert len(parsed) == 1


def test_parse_runs_rejects_a_duplicate_run_id() -> None:
    record = _record()
    with pytest.raises(ValueError, match="duplicate run_id"):
        parse_runs([_line(record), _line(record)])


def test_parse_runs_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_runs(["{oops"])


def test_parse_runs_rejects_a_non_object_line() -> None:
    with pytest.raises(ValueError, match="expected an object"):
        parse_runs(["[1, 2, 3]"])


def test_parse_runs_reports_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_runs(['{"run_id": "r1"}'])


def test_parse_runs_rejects_an_unknown_category() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        parse_runs(['{"run_id":"r1","task_id":"x","category":"weather",'
                    '"writer_level":0,"consumer_level":1,"model_family":"a",'
                    '"valid":true,"inflated":false}'])


def test_load_runs_reads_a_file(tmp_path: Path) -> None:
    records = _synthetic_runs(seed=2, n_per_cell=3)
    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(_line(r) for r in records), encoding="utf-8")
    assert len(load_runs(path)) == len(records)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def test_design_codes_the_registered_factors() -> None:
    design = build_design(valid_runs(_synthetic_runs(seed=3, n_per_cell=15)))
    assert design.names[0] == "intercept"
    # The reference levels are absent as columns, which is what makes the
    # registered contrasts single coefficients or differences of two.
    assert "gap[0->1]" not in design.names
    assert "category[control]" not in design.names
    assert "gap[0->2]" in design.names
    assert "gap[1->2]" in design.names
    # intercept + 2 gap + 4 category + 2 family
    assert design.X.shape[1] == 9


def test_design_marks_each_row_with_its_own_levels() -> None:
    design = build_design(valid_runs(_synthetic_runs(seed=4, n_per_cell=15)))
    gap_02 = design.index("gap[0->2]")
    gap_12 = design.index("gap[1->2]")
    # Treatment coding: no row is in two gap conditions at once, and a row in
    # the reference condition is in neither column.
    assert set(np.unique(design.X[:, gap_02])) <= {0.0, 1.0}
    assert (design.X[:, gap_02] * design.X[:, gap_12]).sum() == 0.0


def test_design_rejects_runs_that_were_discarded() -> None:
    runs = _synthetic_runs(seed=5, n_per_cell=15, discard=0.5)
    with pytest.raises(ValueError, match="valid runs only"):
        build_design(runs)


def test_design_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="no runs"):
        build_design([])


def test_design_rejects_a_sample_without_the_control_category() -> None:
    runs = [r for r in _synthetic_runs(seed=6, n_per_cell=15) if not r.category.is_control]
    with pytest.raises(ValueError, match="control category has no valid runs"):
        build_design(runs)


def test_design_rejects_a_sample_without_the_reference_gap() -> None:
    runs = [r for r in _synthetic_runs(seed=7, n_per_cell=15) if r.gap != (0, 1)]
    with pytest.raises(ValueError, match="reference gap condition"):
        build_design(runs)


def test_contrast_builds_the_vector_it_is_asked_for() -> None:
    design = build_design(valid_runs(_synthetic_runs(seed=8, n_per_cell=15)))
    vector = design.contrast({"gap[0->2]": 1.0, "gap[1->2]": -1.0})
    assert vector[design.index("gap[0->2]")] == 1.0
    assert vector[design.index("gap[1->2]")] == -1.0
    assert vector.sum() == 0.0


def test_design_index_rejects_an_unknown_column() -> None:
    design = build_design(valid_runs(_synthetic_runs(seed=9, n_per_cell=15)))
    with pytest.raises(KeyError):
        design.index("gap[9->9]")


# ---------------------------------------------------------------------------
# Quadrature: against a different quadrature
# ---------------------------------------------------------------------------


def test_adaptive_quadrature_matches_a_dense_fixed_grid() -> None:
    """The check that the estimator's integral is the integral it claims to be."""
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(QUADRATURE_NODES)
    log_weights = np.log(weights)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(100):
        size = int(rng.integers(1, 60))
        eta = rng.normal(-2.0, 1.5, size=size)
        y = (rng.random(size) < 0.25).astype(float)
        sigma = float(rng.uniform(0.01, 3.0))
        adaptive = _group_log_marginal(eta, y, sigma, nodes, log_weights)
        dense = _dense_log_marginal(eta, y, sigma)
        worst = max(worst, abs(adaptive - dense))
    assert worst < 1e-4, f"adaptive quadrature differs from a dense grid by {worst}"


def test_quadrature_survives_a_large_linear_predictor() -> None:
    """A large |eta| must not overflow: the likelihood is computed in log space."""
    from numpy.polynomial.hermite import hermgauss

    nodes, weights = hermgauss(QUADRATURE_NODES)
    eta = np.array([40.0, -40.0, 0.0])
    y = np.array([1.0, 0.0, 1.0])
    value = _group_log_marginal(eta, y, 2.0, nodes, np.log(weights))
    assert math.isfinite(value)


def test_quadrature_is_stable_in_the_node_count() -> None:
    """If 9 and 25 nodes disagree, the node count is doing work it should not."""
    from numpy.polynomial.hermite import hermgauss

    rng = np.random.default_rng(12)
    eta = rng.normal(-2.0, 1.0, size=40)
    y = (rng.random(40) < 0.2).astype(float)
    values = []
    for count in (9, 25, 60):
        nodes, weights = hermgauss(count)
        values.append(_group_log_marginal(eta, y, 0.8, nodes, np.log(weights)))
    assert max(values) - min(values) < 1e-6


# ---------------------------------------------------------------------------
# Estimator: against a different model, and against the sampling distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [13, 14, 15])
def test_fit_reduces_to_ordinary_logistic_when_there_is_no_task_effect(seed: int) -> None:
    """Data simulated with sigma = 0 must come back as ordinary logistic regression.

    This is the check against a different model: scipy finds the plain logistic
    maximum likelihood with no reference to this module, and the mixed fit has
    to land on it once the random effect it adds is estimated away.
    """
    rng = np.random.default_rng(seed)
    y, X, groups = _simulate(
        rng, n_groups=25, per_group=24, beta=np.array([-1.0, 0.8]), sigma=0.0
    )
    fit = fit_random_intercept_logit(y, X, groups)
    mle = _plain_logistic_mle(X, y)
    assert fit.sigma < 0.35, f"a random effect was found where none was simulated: {fit.sigma}"
    assert np.abs(fit.params - mle).max() < 0.05


def test_sigma_at_bound_flags_a_random_effect_that_was_not_estimated() -> None:
    """The flag is reported so a degenerate fit is not read as a converged one."""
    from model import _LOG_SIGMA_BOUNDS, FitResult

    def _fit_with(sigma: float) -> FitResult:
        return FitResult(
            names=("intercept",),
            params=np.zeros(1),
            cov=np.eye(1),
            sigma=sigma,
            loglik=-1.0,
            n_observations=1,
            n_groups=1,
            n_nodes=QUADRATURE_NODES,
            converged=True,
            message="",
        )

    assert _fit_with(math.exp(_LOG_SIGMA_BOUNDS[0])).sigma_at_bound
    assert not _fit_with(0.5).sigma_at_bound


def test_fit_recovers_the_parameters_it_was_simulated_from() -> None:
    beta = np.array([-1.8, 0.7])
    sigma = 0.6
    rng = np.random.default_rng(14)
    estimates = []
    sigmas = []
    for _ in range(_ALT_REPLICATIONS):
        y, X, groups = _simulate(
            rng, n_groups=_MC_GROUPS, per_group=_MC_PER_GROUP, beta=beta, sigma=sigma
        )
        fit = fit_random_intercept_logit(y, X, groups, n_nodes=_MC_NODES)
        estimates.append(fit.params)
        sigmas.append(fit.sigma)
    mean = np.mean(estimates, axis=0)
    assert abs(mean[1] - beta[1]) < 0.2, f"slope recovered as {mean[1]}, simulated {beta[1]}"
    assert abs(mean[0] - beta[0]) < 0.3, f"intercept recovered as {mean[0]}"
    assert 0.2 < float(np.median(sigmas)) < 1.1


def test_wald_type_one_error_and_coverage_under_the_null() -> None:
    """With no true effect the test must reject at its nominal rate, not above it.

    This is the test that would fail on a wrong covariance matrix. Parameter
    recovery would not: a biased standard error leaves the point estimate
    untouched and only shows up in how often the interval misses.
    """
    beta = np.array([-1.5, 0.0])
    rng = np.random.default_rng(15)
    rejections = 0
    covered = 0
    usable = 0
    for _ in range(_NULL_REPLICATIONS):
        y, X, groups = _simulate(
            rng, n_groups=_MC_GROUPS, per_group=_MC_PER_GROUP, beta=beta, sigma=0.5
        )
        fit = fit_random_intercept_logit(y, X, groups, n_nodes=_MC_NODES)
        error = fit.standard_errors[1]
        if not math.isfinite(error) or error <= 0.0:
            continue
        usable += 1
        z = fit.params[1] / error
        if abs(z) > 1.959963984540054:
            rejections += 1
        if abs(fit.params[1] - beta[1]) <= 1.959963984540054 * error:
            covered += 1
    assert usable > 0.9 * _NULL_REPLICATIONS, f"only {usable} fits were usable"
    rate = rejections / usable
    coverage = covered / usable
    assert 0.01 < rate < 0.11, f"type I error {rate} is not near 0.05"
    assert 0.89 < coverage < 1.0, f"Wald coverage {coverage} is not near 0.95"


def test_fit_detects_an_effect_it_is_powered_for() -> None:
    beta = np.array([-1.2, 1.5])
    rng = np.random.default_rng(16)
    rejections = 0
    for _ in range(_ALT_REPLICATIONS):
        y, X, groups = _simulate(
            rng, n_groups=_MC_GROUPS, per_group=_MC_PER_GROUP, beta=beta, sigma=0.5
        )
        fit = fit_random_intercept_logit(y, X, groups, n_nodes=_MC_NODES)
        z = fit.params[1] / fit.standard_errors[1]
        if z > 1.6448536269514722:
            rejections += 1
    assert rejections / _ALT_REPLICATIONS > 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"y": np.array([0.0, 2.0])}, "binary"),
        ({"n_nodes": 2}, "at least 3"),
    ],
)
def test_fit_rejects_bad_input(kwargs: dict, match: str) -> None:
    y = np.array([0.0, 1.0])
    X = np.ones((2, 1))
    groups = np.zeros(2, dtype=int)
    call = {"y": y, "X": X, "groups": groups}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fit_random_intercept_logit(**call)


def test_fit_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same number of runs"):
        fit_random_intercept_logit(
            np.array([0.0, 1.0]), np.ones((3, 1)), np.zeros(2, dtype=int)
        )


def test_fit_rejects_names_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="one entry per column"):
        fit_random_intercept_logit(
            np.array([0.0, 1.0]), np.ones((2, 1)), np.zeros(2, dtype=int), names=("a", "b")
        )


# ---------------------------------------------------------------------------
# Contrasts and multiplicity
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _fitted(seed: int = 21) -> tuple[Design, object]:
    """One fit, reused across the contrast tests, which all read the same one."""
    runs = valid_runs(_synthetic_runs(seed=seed, n_per_cell=30))
    design = build_design(runs)
    fit = fit_random_intercept_logit(
        design.y, design.X, design.groups, names=design.names, n_nodes=_MC_NODES
    )
    return design, fit


def test_h1_states_the_two_registered_comparisons_in_order() -> None:
    design, fit = _fitted()
    results = h1_contrasts(design, fit)
    assert [r.label for r in results] == [
        "MIR(0->2) > MIR(0->1)",
        "MIR(0->2) > MIR(1->2)",
    ]
    assert results[1].hypothesis == "H1 (decisive)"


def test_h2_states_one_comparison_per_substantive_category() -> None:
    design, fit = _fitted()
    results = h2_contrasts(design, fit)
    assert len(results) == H2_COMPARISONS
    assert all("control" not in r.label.split(">")[0] for r in results)


def test_wald_contrast_directions_are_complementary() -> None:
    design, fit = _fitted()
    vector = design.contrast({"gap[0->2]": 1.0})
    greater = wald_contrast(fit, vector, label="x", hypothesis="h", alternative="greater")
    less = wald_contrast(fit, vector, label="x", hypothesis="h", alternative="less")
    assert greater.p_value + less.p_value == pytest.approx(1.0, abs=1e-9)


def test_wald_contrast_rejects_an_unknown_alternative() -> None:
    design, fit = _fitted()
    with pytest.raises(ValueError, match="unknown alternative"):
        wald_contrast(
            fit,
            design.contrast({"gap[0->2]": 1.0}),
            label="x",
            hypothesis="h",
            alternative="sideways",
        )


def _fake(p: float, label: str = "c"):
    from model import ContrastResult

    return ContrastResult(label, "H", 0.0, 1.0, 0.0, p, "greater")


def test_holm_matches_the_textbook_procedure() -> None:
    results = holm([_fake(0.01), _fake(0.02), _fake(0.03)], alpha=0.05)
    # 3 * 0.01 = 0.03; 2 * 0.02 = 0.04; 1 * 0.03 = 0.04 after the monotone step.
    assert [round(r.p_adjusted, 10) for r in results] == [0.03, 0.04, 0.04]
    assert all(r.rejected for r in results)


def test_holm_is_monotone_in_the_sorted_order() -> None:
    results = holm([_fake(0.04), _fake(0.001), _fake(0.5)], alpha=0.05)
    ordered = sorted(results, key=lambda r: r.p_value)
    adjusted = [r.p_adjusted for r in ordered]
    assert adjusted == sorted(adjusted)


def test_holm_stops_at_the_first_failure() -> None:
    results = holm([_fake(0.30), _fake(0.001)], alpha=0.05)
    by_label = {r.p_value: r for r in results}
    assert by_label[0.001].rejected
    assert not by_label[0.30].rejected


def test_holm_caps_adjusted_values_at_one() -> None:
    results = holm([_fake(0.9), _fake(0.95)], alpha=0.05)
    assert all(r.p_adjusted <= 1.0 for r in results)
    assert not any(r.rejected for r in results)


def test_holm_of_a_single_comparison_is_the_comparison() -> None:
    results = holm([_fake(0.04)], alpha=0.05)
    assert results[0].p_adjusted == pytest.approx(0.04)
    assert results[0].rejected


def test_holm_of_nothing_is_nothing() -> None:
    assert holm([]) == ()


# ---------------------------------------------------------------------------
# Descriptive MIR, bootstrap, equivalence
# ---------------------------------------------------------------------------


def test_per_cell_and_per_category_mir_count_valid_runs_only() -> None:
    runs = _synthetic_runs(seed=22, n_per_cell=30, discard=0.3)
    cells = per_cell_mir(runs)
    assert len(cells) == len(GAP_CONDITIONS) * len(Category)
    assert all(0.0 <= v <= 1.0 for v in cells.values())
    categories = per_category_mir(runs)
    assert set(categories) == {c.value for c in Category}


def test_bootstrap_is_reproducible_and_brackets_the_point_estimate() -> None:
    runs = _synthetic_runs(seed=23, n_per_cell=30)
    first = bootstrap_cell_ci(runs, "0->2", Category.FINANCIAL, resamples=400, seed=5)
    second = bootstrap_cell_ci(runs, "0->2", Category.FINANCIAL, resamples=400, seed=5)
    assert first == second
    point, lower, upper = first
    assert lower <= point <= upper
    assert point == pytest.approx(per_cell_mir(runs)[("0->2", "financial")])


def test_bootstrap_resamples_tasks_rather_than_runs() -> None:
    """A run-level bootstrap on clustered data returns an interval that is too narrow.

    The registered interval resamples tasks, so when the outcome is driven by
    the task it must be the wider of the two. If this ever fails, the clustering
    has stopped doing anything.
    """
    runs = _synthetic_runs(seed=24, n_per_cell=30, sigma_task=1.6)
    _, lower, upper = bootstrap_cell_ci(
        runs, "0->2", Category.FINANCIAL, resamples=600, seed=6
    )
    cell = [
        r
        for r in valid_runs(runs)
        if r.gap_label == "0->2" and r.category is Category.FINANCIAL
    ]
    outcomes = np.array([1.0 if r.inflated else 0.0 for r in cell])
    rng = np.random.default_rng(6)
    draws = np.array(
        [outcomes[rng.integers(0, len(outcomes), len(outcomes))].mean() for _ in range(600)]
    )
    naive_lower, naive_upper = np.quantile(draws, [0.025, 0.975])
    assert (upper - lower) > (naive_upper - naive_lower)


def test_bootstrap_rejects_an_empty_cell() -> None:
    runs = [r for r in _synthetic_runs(seed=25, n_per_cell=15) if r.gap_label != "1->2"]
    with pytest.raises(ValueError, match="no valid runs in cell"):
        bootstrap_cell_ci(runs, "1->2", Category.FINANCIAL, resamples=10)


def test_equivalence_uses_the_registered_margin_and_alpha() -> None:
    runs = _synthetic_runs(seed=26, n_per_cell=30)
    results = tost_equivalence(runs)
    assert len(results) == H2_COMPARISONS
    assert all(r.margin == EQUIVALENCE_MARGIN == MDE_REGISTERED for r in results)
    expected_alpha = holm_planning_alpha(ALPHA_FAMILY, H2_COMPARISONS)
    assert all(r.alpha == pytest.approx(expected_alpha) for r in results)
    assert expected_alpha == pytest.approx(0.0125)


def test_equivalence_declares_identical_categories_equivalent() -> None:
    runs = _synthetic_runs(seed=27, category_effect=0.0, n_per_cell=135)
    results = tost_equivalence(runs)
    assert sum(r.equivalent for r in results) >= H2_COMPARISONS - 1


def test_equivalence_refuses_a_category_beyond_the_margin() -> None:
    runs = _synthetic_runs(seed=28, category_effect=1.6, n_per_cell=135)
    results = tost_equivalence(runs)
    assert not any(r.equivalent for r in results)


def test_equivalence_rejects_a_sample_without_a_control() -> None:
    runs = [r for r in _synthetic_runs(seed=29, n_per_cell=15) if not r.category.is_control]
    with pytest.raises(ValueError, match="control category has no valid runs"):
        tost_equivalence(runs)


# ---------------------------------------------------------------------------
# Attribution and discards
# ---------------------------------------------------------------------------


def test_air_is_the_share_of_events_with_a_recovered_causal_set() -> None:
    runs = _synthetic_runs(seed=30, effect_02=1.0, n_per_cell=30, attribution=0.75)
    air = attribution_identification_rate(runs)
    assert 0.5 < air < 0.95


def test_air_is_undefined_without_events() -> None:
    runs = _synthetic_runs(seed=31, n_per_cell=15, intercept=-40.0)
    assert math.isnan(attribution_identification_rate(runs))


def test_discard_rate_counts_invalid_runs() -> None:
    runs = _synthetic_runs(seed=32, n_per_cell=30, discard=0.25)
    assert discard_rate(runs) == pytest.approx(0.25, abs=0.06)
    assert discard_rate([]) == 0.0


# ---------------------------------------------------------------------------
# The pre-committed criteria of §6
# ---------------------------------------------------------------------------


def test_f1_triggers_when_the_gap_does_nothing() -> None:
    verdict = evaluate_falsification(
        _synthetic_runs(seed=33, effect_02=0.0, effect_12=0.0, n_per_cell=135),
        n_nodes=_MC_NODES,
    )
    assert verdict.f1_triggered


def test_f1_does_not_trigger_on_a_clear_ordered_effect() -> None:
    verdict = evaluate_falsification(
        _synthetic_runs(seed=34, effect_02=1.4, effect_12=0.2, n_per_cell=135),
        n_nodes=_MC_NODES,
    )
    assert not verdict.f1_triggered


def test_a_capability_count_effect_triggers_f1_and_is_named() -> None:
    """§6: significance confined to the confounded comparison is not support for H1.

    Here the two conditions with an L2 consumer behave identically and only the
    L0->L1 condition is lower, which is exactly the pattern a capability-count
    effect produces.
    """
    verdict = evaluate_falsification(
        _synthetic_runs(seed=35, effect_02=1.4, effect_12=1.4, n_per_cell=135),
        n_nodes=_MC_NODES,
    )
    assert verdict.f1_triggered
    assert any("capability-count" in note for note in verdict.notes)


def test_f2_triggers_when_every_category_matches_the_control() -> None:
    verdict = evaluate_falsification(
        _synthetic_runs(seed=36, effect_02=1.4, category_effect=0.0, n_per_cell=135),
        n_nodes=_MC_NODES,
    )
    assert verdict.f2_triggered == all(r.equivalent for r in verdict.equivalence)


def test_f3_triggers_on_a_low_attribution_rate() -> None:
    verdict = evaluate_falsification(
        _synthetic_runs(seed=37, effect_02=1.0, n_per_cell=60, attribution=0.2),
        n_nodes=_MC_NODES,
    )
    assert verdict.air < AIR_F3_THRESHOLD
    assert verdict.f3_triggered


def test_f4_triggers_above_the_discard_ceiling() -> None:
    verdict = evaluate_falsification(
        _synthetic_runs(seed=38, effect_02=1.0, n_per_cell=60, discard=0.45),
        n_nodes=_MC_NODES,
    )
    assert verdict.f4_triggered


def test_evaluation_rejects_a_sample_with_nothing_valid() -> None:
    runs = _synthetic_runs(seed=39, n_per_cell=15, discard=1.0)
    with pytest.raises(ValueError, match="every run was discarded"):
        evaluate_falsification(runs)


def test_evaluation_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="no runs to analyse"):
        evaluate_falsification([])


def test_report_renders_every_registered_section() -> None:
    text = report(_synthetic_runs(seed=40, effect_02=1.0, effect_12=0.3, n_per_cell=30))
    for fragment in (
        "mixed-effects logistic regression",
        "sigma (task random intercept)",
        "MIR(0->2) > MIR(1->2)",
        "per-cell MIR (descriptive only",
        "pre-committed falsification criteria",
        "AIR =",
    ):
        assert fragment in text
